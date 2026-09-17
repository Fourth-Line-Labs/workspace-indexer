import { Thing } from './models';

export class Store {
  find(id: string): Thing {
    return { id };
  }
}
